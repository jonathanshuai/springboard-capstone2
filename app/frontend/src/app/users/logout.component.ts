import {Component, OnDestroy, OnInit} from '@angular/core';
import {UsersApiService} from "./users-api.service";
import {Router} from "@angular/router";

@Component({
  selector: 'login-form',
  template: 'Logging out...',  
  styleUrls: ['../app.component.css']
})
export class LogoutComponent implements OnInit{

  constructor(private usersApi: UsersApiService, private router: Router) { }

  ngOnInit() {
    this.usersApi.logout();
    this.router.navigate(['/login']);
  }
}
