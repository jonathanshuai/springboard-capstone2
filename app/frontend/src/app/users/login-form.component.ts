import {Component} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {UsersApiService} from "./users-api.service";
import {Router} from "@angular/router";

import { NgModule }      from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule }   from '@angular/forms';
import { FormGroup, FormControl } from '@angular/forms';

import { Directive, forwardRef, Attribute } from '@angular/core';
import { Validators, Validator, AbstractControl, NG_VALIDATORS } from '@angular/forms';

@Component({
  selector: 'login-form',
  templateUrl: './login-form.component.html',  
  styleUrls: ['../app.component.css']
})
export class LoginFormComponent {
  user = {
    username: '',
    password: ''
  };

  serverError = false;
  serverErrorMessage = '';
  updateServerError(event: any) { this.serverError = false; }
  constructor(private usersApi: UsersApiService, private router: Router) { }

  authenticateUser() {
    this.user = {
      username: this.loginForm.get('username').value,
      password: this.loginForm.get('password').value
    }

    this.usersApi
      .authenticateUser(this.user)
      .subscribe(
        () => this.router.navigate(['/']),
        error => {
          this.serverError = true; 
          this.serverErrorMessage = error.error;
          console.log(error)
        });
  }

  loginForm = new FormGroup({
    'username': new FormControl('', [Validators.required]),
    'password': new FormControl('', [Validators.required])
  });

  get username() { return this.loginForm.get('username'); }
  get password() { return this.loginForm.get('password'); }
}
