import {Component, OnInit} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Router} from "@angular/router";

import { NgModule }      from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule }   from '@angular/forms';
import { FormGroup, FormControl } from '@angular/forms';

import {UsersApiService} from '../users/users-api.service';
import {RestrictionsApiService} from './restrictions-api.service';

@Component({
  selector: 'restrictions',
  templateUrl: './restrictions.component.html',  
  styleUrls: ['../app.component.css']
})
export class RestrictionsComponent{
  constructor(private usersApi: UsersApiService, 
              private restrictionsApi: RestrictionsApiService, 
              private router: Router) { }

  restriction = {
    vegan: false,
    vegetarian: false,
    peanut_free: false
  }

  ngOnInit(){
    if (!this.usersApi.loggedIn()){
      this.router.navigate(['/login'])
    }
    else{
      this.getRestrictions();
    }
  }

  getRestrictions() {
    this.restrictionsApi
      .getRestrictions()
      .subscribe(
        result => {
          this.restriction.vegan = result['vegan']
          this.restriction.vegetarian = result['vegetarian']
          this.restriction.peanut_free = result['peanut_free']
        },
        error => console.log('error')
      );
  }

  saveRestrictions(){
    console.log(this.restriction)
    this.restrictionsApi
      .saveRestrictions(this.restriction)
      .subscribe(
        result => {
          console.log(result);
          this.router.navigate(['/']);
        },
        error => console.log('error')
      );
  }
}
