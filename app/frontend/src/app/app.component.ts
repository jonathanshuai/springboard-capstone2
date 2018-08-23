import {Component, ViewEncapsulation} from '@angular/core';
import {Router, NavigationEnd} from "@angular/router";
import {Subscription} from 'rxjs/Subscription';

@Component({
  selector: 'app-root',
  encapsulation: ViewEncapsulation.None,
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent { 

  constructor(private router: Router) { }

  navbarVisible = false;
 
  ngOnInit(){
      this.router.events
      .subscribe((event) => {
        // Make navbar visible if we're not on register or login
        if (event instanceof NavigationEnd){
            if (event['url'] == '/register' || event['url'] == '/login'){
                this.navbarVisible = false;
            }
            else{
                this.navbarVisible = true;
            }
        } 
      });
  }

}