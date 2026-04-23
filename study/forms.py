from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import Subject


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'placeholder': 'your@email.com',
            'autocomplete': 'email'
        })
    )
    first_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'First name'})
    )
    last_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'Last name'})
    )

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'username', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'placeholder': 'Username'})
        self.fields['password1'].widget.attrs.update({'placeholder': 'Password'})
        self.fields['password2'].widget.attrs.update({'placeholder': 'Confirm Password'})
        for field in self.fields.values():
            field.widget.attrs['class'] = 'auth-input'

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
            # Create default subjects for new user
            default_subjects = [
                {'name': 'Đại cương', 'icon': '📚'},
                {'name': 'Chuyên Ngành', 'icon': '🎓'},
            ]
            for s in default_subjects:
                Subject.objects.create(owner=user, name=s['name'], icon=s['icon'])
        return user


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'placeholder': 'Username',
            'class': 'auth-input'
        })
        self.fields['password'].widget.attrs.update({
            'placeholder': 'Password',
            'class': 'auth-input'
        })


class SubjectForm(forms.ModelForm):
    ICON_OPTIONS = [
        ('📚', '📚 Đại cương'),
        ('🎓', '🎓 Chuyên Ngành'),
        ('⭐', '⭐ Tự chọn'),
    ]
    icon = forms.ChoiceField(choices=ICON_OPTIONS, widget=forms.Select(attrs={'class': 'auth-input'}))

    class Meta:
        model = Subject
        fields = ['name', 'icon']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'auth-input', 'placeholder': 'Subject name'})
        }
