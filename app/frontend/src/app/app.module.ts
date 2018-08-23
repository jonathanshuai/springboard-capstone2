import {BrowserModule} from '@angular/platform-browser';
import {NgModule} from '@angular/core';
import {FormsModule}   from '@angular/forms';
import {ReactiveFormsModule} from '@angular/forms';
import {HttpClientModule} from '@angular/common/http';

import { MatProgressSpinnerModule } from '@angular/material';
import { MatCheckboxModule } from '@angular/material/checkbox';

import {UsersApiService} from './users/users-api.service';
import {RecipesApiService} from './recipes/recipes-api.service';
import {RestrictionsApiService} from './restrictions/restrictions-api.service';

import {RouterModule, Routes} from '@angular/router';

import {AppComponent} from './app.component';
import {RegisterFormComponent} from './users/register-form.component';
import {LoginFormComponent} from './users/login-form.component';
import {LogoutComponent} from './users/logout.component';
import {RecommenderComponent} from './recipes/recommender.component';
import {RecipesComponent} from './recipes/recipes.component';
import {RestrictionsComponent} from './restrictions/restrictions.component';

import { HTTP_INTERCEPTORS } from '@angular/common/http';
import { TokenInterceptor } from './users/token.interceptor';

const appRoutes: Routes = [
  { path: 'register', component: RegisterFormComponent },
  { path: 'login', component: LoginFormComponent },
  { path: 'logout', component: LogoutComponent },
  { path: '', component: RecommenderComponent},
  { path: 'recipes', component: RecipesComponent },
  { path: 'settings', component: RestrictionsComponent },
];

@NgModule({
  declarations: [
    AppComponent,
    RegisterFormComponent,
    LoginFormComponent,
    LogoutComponent,
    RecommenderComponent,
    RestrictionsComponent,
    RecipesComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    MatProgressSpinnerModule,
    MatCheckboxModule,
    RouterModule.forRoot(
      appRoutes,
    ),
  ],
  providers: [
    UsersApiService, 
    RecipesApiService, 
    RestrictionsApiService,
    {
      provide: HTTP_INTERCEPTORS,
      useClass: TokenInterceptor,
      multi: true
    } 
  ],
  bootstrap: [AppComponent],  
})
export class AppModule {
    
}
