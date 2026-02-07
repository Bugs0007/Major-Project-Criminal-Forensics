from django.contrib import admin
from .models import FaceImage


# @admin.ModelAdmin
class FaceImageAdmin(admin.ModelAdmin):
    list_display = ['name', 'original_filename', 'uploaded_at']
    list_filter = ['uploaded_at']
    search_fields = ['name', 'original_filename', 'tags']
    readonly_fields = ['uploaded_at', 'updated_at', 'image_url']
    
    fieldsets = (
        ('Image Information', {
            'fields': ('image_url', 'original_filename')
        }),
        ('Metadata', {
            'fields': ('name', 'tags', 'notes')
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'updated_at')
        }),
    )


admin.site.register(FaceImage, FaceImageAdmin)