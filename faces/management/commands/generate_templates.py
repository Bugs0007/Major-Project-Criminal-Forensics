from django.core.management.base import BaseCommand
from faces.feature_templates import FeatureTemplateGenerator
from django.conf import settings
import os


class Command(BaseCommand):
    help = 'Generate facial feature templates'

    def handle(self, *args, **options):
        output_dir = os.path.join(settings.MEDIA_ROOT, 'feature_templates')
        FeatureTemplateGenerator.save_templates_to_disk(output_dir)
        self.stdout.write(
            self.style.SUCCESS(f'Successfully generated templates in {output_dir}')
        )