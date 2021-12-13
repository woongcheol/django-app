from django.shortcuts import render
from django.views.generic import CreateView
from profileapp.forms import ProfileCreationForm

from profileapp.models import Profile
from django.urls import reverse_lazy

# Create your views here.


class ProfileCreateView(CreateView):
    model = Profile
    cotext_object_name = 'target_profile'
    form_class = ProfileCreationForm
    success_url = reverse_lazy('predict:food')
    template_name = 'profileapp/create.html'

    def form_valid(self, form):
        temp_profile = form.save(commit=False)
        temp_profile.user = self.request.user
        temp_profile.save()
        return super().form_valid(form)