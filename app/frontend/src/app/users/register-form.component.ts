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
  selector: 'register-form',
  templateUrl: './register-form.component.html',  
  styleUrls: ['../app.component.css']
})
export class RegisterFormComponent {
  user = {
    username: '',
    password: '',
    name: '',
  };

  serverError = false;
  serverErrorMessage = '';
  updateServerError(event: any) { this.serverError = false; }
  constructor(private usersApi: UsersApiService, private router: Router) { }

  registerUser() {
    this.user = {
      username: this.registerForm.get('username').value,
      password: this.registerForm.get('password').value,
      name: this.registerForm.get('name').value
    }

    this.usersApi
      .registerUser(this.user)
      .subscribe(
        () => this.router.navigate(['/login']),
        error => {
          this.serverError = true; 
          this.serverErrorMessage = error.error;
        });
  }

  registerForm = new FormGroup({
    'username': new FormControl('', [Validators.required]),
    'password': new FormControl('', [Validators.required]),
    'passwordconf': new FormControl('', [Validators.required, matchOtherValidator('password')]),
    'name': new FormControl('')
  });

  get username() { return this.registerForm.get('username'); }
  get password() { return this.registerForm.get('password'); }
  get passwordconf() { return this.registerForm.get('passwordconf'); }
  get name() { return this.registerForm.get('name'); }
}

// Thanks to https://gist.github.com/slavafomin/17ded0e723a7d3216fb3d8bf845c2f30
export function matchOtherValidator (otherControlName: string) {
  let thisControl: FormControl;
  let otherControl: FormControl;

  return function matchOtherValidate (control: FormControl) {

    if (!control.parent) {
      return null;
    }

    // Initializing the validator.
    if (!thisControl) {
      thisControl = control;
      otherControl = control.parent.get(otherControlName) as FormControl;
      if (!otherControl) {
        throw new Error('matchOtherValidator(): other control is not found in parent group');
      }
      otherControl.valueChanges.subscribe(() => {
        thisControl.updateValueAndValidity();
      });
    }

    if (!otherControl) {
      return null;
    }

    if (otherControl.value !== thisControl.value) {
      return {
        matchOther: true
      };
    }

    return null;

  }

}