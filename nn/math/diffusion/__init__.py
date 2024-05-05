from .schedulers import SchedulerConfig
from diffusers import DDPMScheduler

STRING_TO_SCHEDULER = {
    "DDPMScheduler":DDPMScheduler,
}