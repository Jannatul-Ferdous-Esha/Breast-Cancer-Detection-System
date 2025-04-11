from django.shortcuts import render, HttpResponseRedirect
from App_Login.forms import CreateNewUser, EditProfile
from django.contrib.auth import authenticate, login, logout  
from django.urls import reverse, reverse_lazy
from App_Login.models import UserProfile
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required


# Create your views here.
def sign_up(request):
    form = CreateNewUser()
    registered = False
    if request.method == 'POST':
        form = CreateNewUser(data=request.POST)
        if form.is_valid():
            user = form.save()
            registered = True
            user_profile = UserProfile(user=user)
            user_profile.save()
            #login(request, user)  # Log in the user after signup
            return HttpResponseRedirect(reverse('App_Login:login' ))  # Redirect to a relevant page

    return render(request, 'App_Login/signup.html', context={'title': 'Signup Form Here', 'form': form})  

def login_page(request):
    form = AuthenticationForm()
    if request. method == 'POST':
        form = AuthenticationForm(data = request.POST)
        if form.is_valid():
            username =form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user =authenticate(username=username, password=password)
            if user is not None:
              login(request,user) 
              return HttpResponseRedirect(reverse('App_Login:edit'))
    
    return render(request,'App_Login/login.html',context={'title' :'Login Page', 'form' : form})       
            
@login_required
def edit_profile(request):
    current_user = UserProfile.objects.get(user=request.user)
    form = EditProfile(instance = current_user)
    return render(request, 'App_Login/profile.html',context = {'title':'Edit Profile Page','form': form})    
    

@login_required
def logout_user(request):
    logout(request)
    return HttpResponseRedirect(reverse('App_Login:login'))