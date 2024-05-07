from .schedulers import SchedulerConfig, DiffusersScheduler, SeisFusionScheduler

STRING_TO_SCHEDULER = {
    "DDPMScheduler":DiffusersScheduler,
    "SeisFusionScheduler":SeisFusionScheduler,
}