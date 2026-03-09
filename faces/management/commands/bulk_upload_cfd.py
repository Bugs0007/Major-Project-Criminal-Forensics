import os
import glob
from django.core.management.base import BaseCommand
from django.conf import settings
from faces.models import FaceImage
from faces.utils import upload_to_s3, get_face_encoding, resize_image_if_needed
from django.core.files.uploadedfile import SimpleUploadedFile


class Command(BaseCommand):
    help = 'Bulk upload face images from the CFD dataset into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dir',
            type=str,
            default=os.path.join(
                settings.BASE_DIR, 'cfd', 'CFD Version 3.0', 'Images'
            ),
            help='Root directory of CFD images',
        )
        parser.add_argument(
            '--neutral-only',
            action='store_true',
            default=True,
            help='Upload only neutral expression images (filenames ending in -N.jpg)',
        )
        parser.add_argument(
            '--all-expressions',
            action='store_true',
            default=False,
            help='Upload all expression images, not just neutral',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            default=False,
            help='List images that would be uploaded without actually uploading',
        )

    def handle(self, *args, **options):
        root_dir = options['dir']
        neutral_only = not options['all_expressions']
        dry_run = options['dry_run']

        if not os.path.isdir(root_dir):
            self.stderr.write(self.style.ERROR(f'Directory not found: {root_dir}'))
            return

        # Collect all jpg images recursively
        image_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.lower().endswith('.jpg'):
                    continue
                if neutral_only and not fname.upper().endswith('-N.JPG'):
                    continue
                image_paths.append(os.path.join(dirpath, fname))

        image_paths.sort()
        total = len(image_paths)
        self.stdout.write(f'Found {total} images to upload.')

        if dry_run:
            for p in image_paths:
                self.stdout.write(f'  {os.path.basename(p)}')
            return

        success = 0
        skipped = 0
        failed = 0

        for idx, image_path in enumerate(image_paths, 1):
            filename = os.path.basename(image_path)

            # Skip if already uploaded
            if FaceImage.objects.filter(original_filename=filename).exists():
                self.stdout.write(f'[{idx}/{total}] SKIP (exists): {filename}')
                skipped += 1
                continue

            try:
                # Derive a human-readable name from the filename
                # e.g. CFD-AF-200-228-N.jpg -> AF-200
                parts = filename.replace('.jpg', '').replace('.JPG', '').split('-')
                name = '-'.join(parts[1:3]) if len(parts) >= 3 else filename

                with open(image_path, 'rb') as f:
                    image_data = f.read()

                image_file = SimpleUploadedFile(
                    name=filename,
                    content=image_data,
                    content_type='image/jpeg',
                )

                # Resize if needed
                image_file = resize_image_if_needed(image_file)

                # Get face encoding
                encoding = get_face_encoding(image_file)
                if encoding is None:
                    self.stdout.write(
                        self.style.WARNING(f'[{idx}/{total}] NO FACE: {filename}')
                    )
                    failed += 1
                    continue

                # Reset pointer for S3 upload
                image_file.seek(0)

                # Upload to S3
                image_url, s3_filename = upload_to_s3(
                    image_file, filename=f'faces/{filename}'
                )

                # Save to database
                face = FaceImage(
                    image_url=image_url,
                    original_filename=filename,
                    name=name,
                    tags=['CFD', 'dataset'],
                    notes='Bulk uploaded from CFD dataset',
                )
                face.set_encoding(encoding)
                face.save()

                success += 1
                self.stdout.write(
                    self.style.SUCCESS(f'[{idx}/{total}] OK: {filename}')
                )

            except Exception as e:
                failed += 1
                self.stderr.write(
                    self.style.ERROR(f'[{idx}/{total}] FAIL: {filename} — {e}')
                )

        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS(f'Done! {success} uploaded, {skipped} skipped, {failed} failed.'))
