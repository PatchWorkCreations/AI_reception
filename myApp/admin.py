from django.contrib import admin, messages
from django.urls import path, reverse
from django.shortcuts import redirect, render
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST
from django.contrib.admin.views.decorators import staff_member_required
from .twilio_utils import place_outbound_test_call

# --- Admin views (staff-only) ---

@staff_member_required
def tools_home(request):
    # You can pass anything else you want to show on the page via context
    return render(request, "admin/tools.html", {})

@staff_member_required
@require_POST
def tools_call_me(request):
    try:
        sid = place_outbound_test_call()
        messages.success(
            request,
            mark_safe(f"📞 Calling your phone now. Twilio Call SID: <code>{sid}</code>")
        )
    except Exception as e:
        messages.error(request, f"Failed to start call: {e}")
    # send back to the tools page
    return redirect(reverse("admin:neuromed_tools"))

# --- Wire custom URLs into default admin site ---

def _custom_admin_urls():
    return [
        path("tools/", admin.site.admin_view(tools_home), name="neuromed_tools"),
        path("tools/call-me/", admin.site.admin_view(tools_call_me), name="neuromed_tools_call_me"),
    ]

# prepend our URLs to the built-in admin URLs
admin_urls_orig = admin.site.get_urls
def admin_urls_override():
    return _custom_admin_urls() + admin_urls_orig()
admin.site.get_urls = admin_urls_override
