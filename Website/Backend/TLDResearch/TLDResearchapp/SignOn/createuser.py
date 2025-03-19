from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login

def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Automatically logs in the user
            return redirect("login")  # Redirect to login or dashboard
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})