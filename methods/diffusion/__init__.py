from .schedulers import SeisFusionScheduler, SchedulerConfig
from diffusers import DDPMScheduler

STRING_TO_SCHEDULER = {
    "SeisFusionScheduler":SeisFusionScheduler,
    "DDPMScheduler":DDPMScheduler,
}