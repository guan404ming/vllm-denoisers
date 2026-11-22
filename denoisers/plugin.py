from denoisers import patches
from denoisers.models import families


def register() -> None:
    patches.apply()
    families.register_all()
