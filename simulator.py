"""
schedulers.py
=============
Discrete-time simulation of three scheduling algorithms:
  1. Round-Robin (RR)       — general-purpose, no deadline awareness
  2. Earliest Deadline First (EDF) — optimal real-time, dynamic priority
  3. Adaptive Feedback-Controlled EDF (AFC-EDF) — EDF + PI control loop
                                                   that adjusts task budgets
                                                   when deadline misses occur

All schedulers share a common base class and produce a timeline of
ScheduleEvent objects that can be consumed by the visualisation layer.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from math import gcd
from typing import List, Optional, Tuple


#
# Data structures
#

@dataclass
class Task:
    name: str
    period: int
    wcet: int          # worst-case execution time (nominal budget)
    deadline: int      # relative deadline (≤ period for constrained tasks)


@dataclass(order=True)
class JobInstance:
    """A single activation (job) of a periodic task."""
    absolute_deadline: int
    release_time: int = field(compare=False)
    task_name: str    = field(compare=False)
    remaining: int    = field(compare=False)   # execution units still needed
    budget: float     = field(compare=False)   # effective budget (AFC-EDF may shrink this)


@dataclass
class ScheduleEvent:
    time: int
    task_name: str
    deadline_miss: bool = False


def hyperperiod(tasks: List[Task]) -> int:
    """LCM of all task periods."""
    result = tasks[0].period
    for t in tasks[1:]:
        result = result * t.period // gcd(result, t.period)
    return result


#
# Base scheduler
#
class BaseScheduler:
    def __init__(self, tasks: List[Task], sim_duration: int):
        self.tasks        = tasks
        self.sim_duration = sim_duration
        self.timeline: List[ScheduleEvent] = []
        self.deadline_misses = 0
        self.idle_time       = 0
        # per-job response times (for jitter analysis)
        self.response_times: List[float] = []

    def _generate_jobs(self) -> List[JobInstance]:
        jobs = []
        for task in self.tasks:
            t = 0
            while t < self.sim_duration:
                jobs.append(JobInstance(
                    absolute_deadline = t + task.deadline,
                    release_time      = t,
                    task_name         = task.name,
                    remaining         = task.wcet,
                    budget            = float(task.wcet),
                ))
                t += task.period
        jobs.sort(key=lambda j: j.release_time)
        return jobs

    def run(self):
        raise NotImplementedError

    def stats(self) -> dict:
        cpu_used = self.sim_duration - self.idle_time
        return {
            "scheduler":       type(self).__name__,
            "deadline_misses": self.deadline_misses,
            "cpu_utilization": cpu_used / self.sim_duration,
            "idle_time":       self.idle_time,
            "avg_response":    (sum(self.response_times) / len(self.response_times)
                                if self.response_times else 0),
            "jitter":          (max(self.response_times) - min(self.response_times)
                                if len(self.response_times) > 1 else 0),
        }


#
# 1. Round-Robin Scheduler
#

class RoundRobinScheduler(BaseScheduler):
    """
    General-purpose Round-Robin with a fixed time quantum.
    Tasks are served in FIFO order; no deadline awareness whatsoever.
    """

    def __init__(self, tasks: List[Task], sim_duration: int, quantum: int = 2):
        super().__init__(tasks, sim_duration)
        self.quantum = quantum

    def run(self):
        all_jobs   = self._generate_jobs()
        ready: deque[JobInstance] = deque()
        job_idx    = 0
        n          = len(all_jobs)
        current    = None
        q_left     = 0
        missed_ids = set()

        for t in range(self.sim_duration):
            # Release new jobs
            while job_idx < n and all_jobs[job_idx].release_time == t:
                ready.append(all_jobs[job_idx])
                job_idx += 1

            # Quantum expired or no current job → pick next from queue
            if current is None or q_left == 0:
                if current is not None and current.remaining > 0:
                    ready.append(current)
                current = ready.popleft() if ready else None
                q_left  = self.quantum

            # Detect deadline misses (count each job once)
            missed = False
            candidates = list(ready) + ([current] if current else [])
            for job in candidates:
                jid = (job.task_name, job.release_time)
                if job.absolute_deadline <= t and job.remaining > 0 and jid not in missed_ids:
                    missed = True
                    self.deadline_misses += 1
                    missed_ids.add(jid)

            if current:
                current.remaining -= 1
                q_left -= 1
                self.timeline.append(ScheduleEvent(t, current.task_name, missed))
                if current.remaining == 0:
                    self.response_times.append(t + 1 - current.release_time)
                    current = None
                    q_left  = 0
            else:
                self.timeline.append(ScheduleEvent(t, "IDLE"))
                self.idle_time += 1


#
# 2. EDF Scheduler
#

class EDFScheduler(BaseScheduler):
    """
    Earliest Deadline First — optimal preemptive dynamic-priority scheduler.
    At every time unit the ready job with the smallest absolute deadline runs.
    """

    def run(self):
        all_jobs   = self._generate_jobs()
        heap: List[JobInstance] = []
        job_idx    = 0
        n          = len(all_jobs)
        missed_ids = set()

        for t in range(self.sim_duration):
            while job_idx < n and all_jobs[job_idx].release_time == t:
                heapq.heappush(heap, all_jobs[job_idx])
                job_idx += 1

            missed = False
            for job in heap:
                jid = (job.task_name, job.release_time)
                if job.absolute_deadline <= t and job.remaining > 0 and jid not in missed_ids:
                    missed = True
                    self.deadline_misses += 1
                    missed_ids.add(jid)

            if heap:
                current = heapq.heappop(heap)
                current.remaining -= 1
                self.timeline.append(ScheduleEvent(t, current.task_name, missed))
                if current.remaining == 0:
                    self.response_times.append(t + 1 - current.release_time)
                else:
                    heapq.heappush(heap, current)
            else:
                self.timeline.append(ScheduleEvent(t, "IDLE"))
                self.idle_time += 1


#
# 3. AFC-EDF: Adaptive Feedback-Controlled EDF
#

class AFCEDFScheduler(BaseScheduler):
    """
    Adaptive Feedback-Controlled EDF (AFC-EDF).

    Architecture

    • Base layer  : standard EDF priority queue (earliest absolute deadline first).
    • Monitor     : at the end of every observation window W, compute the
                    deadline miss ratio  ρ = misses / jobs_completed_in_window.
    • Controller  : a PI (Proportional-Integral) controller computes a budget
                    scaling factor  α ∈ [α_min, 1.0] such that:
                        α(k+1) = α(k) − Kp·e(k) − Ki·∫e
                    where  e(k) = ρ(k) − ρ_target  (error signal).
    • Actuator    : each newly released job's execution budget is set to
                        budget = max(wcet_min, round(wcet · α))
                    This reduces the CPU demand when the system is overloaded,
                    trading off some computation quality for deadline compliance
                    (a common technique in elastic / imprecise computation).

    Parameters

    window      : observation window length (time units)
    rho_target  : target deadline miss ratio (e.g. 0.05 = 5 %)
    Kp, Ki      : PI controller gains
    alpha_min   : minimum budget scaling factor (floor to avoid starvation)
    """

    def __init__(
        self,
        tasks: List[Task],
        sim_duration: int,
        window: int   = 20,
        rho_target: float = 0.0,
        Kp: float     = 0.3,
        Ki: float     = 0.05,
        alpha_min: float = 0.5,
    ):
        super().__init__(tasks, sim_duration)
        self.window     = window
        self.rho_target = rho_target
        self.Kp         = Kp
        self.Ki         = Ki
        self.alpha_min  = alpha_min

        # Controller state
        self.alpha       = 1.0   # current budget scaling factor
        self.integral    = 0.0   # integral term accumulator

        # Telemetry (for plotting the control loop behaviour)
        self.alpha_history:     List[Tuple[int, float]] = []   # (time, alpha)
        self.miss_ratio_history: List[Tuple[int, float]] = []  # (time, rho)

    def run(self):
        # Build a mutable budget map so we can scale budgets per task
        budget_map = {task.name: float(task.wcet) for task in self.tasks}
        wcet_map   = {task.name: task.wcet        for task in self.tasks}

        # Generate all jobs with nominal budgets first
        all_jobs   = self._generate_jobs()
        heap: List[JobInstance] = []
        job_idx    = 0
        n          = len(all_jobs)
        missed_ids = set()

        # Window counters
        w_jobs_released  = 0
        w_misses         = 0

        for t in range(self.sim_duration):
            # End-of-window: run the PI controller
            if t > 0 and t % self.window == 0:
                rho = w_misses / max(w_jobs_released, 1)
                self.miss_ratio_history.append((t, rho))

                error          = rho - self.rho_target
                self.integral += error
                delta          = self.Kp * error + self.Ki * self.integral
                self.alpha     = max(self.alpha_min, min(1.0, self.alpha - delta))
                self.alpha_history.append((t, self.alpha))

                # Update effective budgets for all tasks
                for task in self.tasks:
                    new_budget = max(1, round(wcet_map[task.name] * self.alpha))
                    budget_map[task.name] = float(new_budget)

                # Reset window counters
                w_jobs_released = 0
                w_misses        = 0

            #  Release new jobs with current effective budget
            while job_idx < n and all_jobs[job_idx].release_time == t:
                job = all_jobs[job_idx]
                job.remaining = max(1, round(budget_map[job.task_name]))
                job.budget    = job.remaining
                heapq.heappush(heap, job)
                job_idx += 1
                w_jobs_released += 1

            #  Detect deadline misses
            missed = False
            for job in heap:
                jid = (job.task_name, job.release_time)
                if job.absolute_deadline <= t and job.remaining > 0 and jid not in missed_ids:
                    missed = True
                    self.deadline_misses += 1
                    missed_ids.add(jid)
                    w_misses += 1

            #  Execute highest-priority (earliest deadline) job
            if heap:
                current = heapq.heappop(heap)
                current.remaining -= 1
                self.timeline.append(ScheduleEvent(t, current.task_name, missed))
                if current.remaining == 0:
                    self.response_times.append(t + 1 - current.release_time)
                else:
                    heapq.heappush(heap, current)
            else:
                self.timeline.append(ScheduleEvent(t, "IDLE"))
                self.idle_time += 1

        # Record final alpha
        self.alpha_history.append((self.sim_duration, self.alpha))




if __name__ == "__main__":
    import copy

    TASKS = [
        Task("T1", period=5,  wcet=3, deadline=3),
        Task("T2", period=8,  wcet=2, deadline=8),
        Task("T3", period=12, wcet=2, deadline=10),
    ]

    H = hyperperiod(TASKS)
    U = sum(t.wcet / t.period for t in TASKS)
    print(f"Task set  |  U = {U:.3f}  |  Hyperperiod = {H}")
    print()

    rr  = RoundRobinScheduler(copy.deepcopy(TASKS), H * 3, quantum=2)
    edf = EDFScheduler(copy.deepcopy(TASKS), H * 3)
    afc = AFCEDFScheduler(copy.deepcopy(TASKS), H * 3, window=20, rho_target=0.0, Kp=0.4, Ki=0.08)

    for sched in [rr, edf, afc]:
        sched.run()
        s = sched.stats()
        print(f"{s['scheduler']:<22}  misses={s['deadline_misses']:3d}  "
              f"util={s['cpu_utilization']:.1%}  "
              f"avg_resp={s['avg_response']:.2f}  jitter={s['jitter']:.2f}")
