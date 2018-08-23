import {Component, OnDestroy, OnInit} from '@angular/core';
import {Subscription} from 'rxjs/Subscription';
import {Router} from "@angular/router";

import {Recipe} from './recipe.model';
import {RecipesApiService} from './recipes-api.service';
import {UsersApiService} from '../users/users-api.service';

import {DomSanitizer} from '@angular/platform-browser';


@Component({
  selector: 'recipes',
  templateUrl: `./recipes.component.html`
})
export class RecipesComponent implements OnInit{
  constructor(private recipesApi: RecipesApiService, 
              private usersApi: UsersApiService, 
              private router: Router,
              private sanitizer: DomSanitizer) { }
  
  recipesList = []

  ngOnInit(){
    if (!this.usersApi.loggedIn()){
      this.router.navigate(['/login'])
    }
    else{
      this.getRecipes();
    }
  }

  getRecipes() {
    this.recipesApi
      .getRecipes()
      .subscribe(
        result => {
          this.recipesList = result;
          for (let recipe of this.recipesList){
            // recipe['url'] = recipe.url;
            recipe['imgsrcsafe'] = this.sanitizer.bypassSecurityTrustStyle(`url(${recipe.imgsrc})`);
            console.log(this.recipesList)
          }
        },
        error => console.log('error')
      );
  }

  toggleRecipe(recipe, event){
    // Hacky
    if (event.srcElement.innerHTML == 'Save Recipe'){
      this.recipesApi
      .saveRecipe(recipe)
      .subscribe(
        result => {
          console.log(result);
        },
        error => console.log(error)
      );
      event.srcElement.innerHTML = 'Delete Recipe';
    }
    else{
      this.recipesApi
      .deleteRecipe(recipe)
      .subscribe(
        result => {
          console.log(event.srcElement);
          console.log(result);
        },
        error => console.log(error)
      );
      event.srcElement.innerHTML = 'Save Recipe';

    }
  }

}