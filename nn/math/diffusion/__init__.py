from .schedulers import SchedulerConfig, DiffusersScheduler

STRING_TO_SCHEDULER = {
    "DDPMScheduler":DiffusersScheduler,
}