import {Component, Renderer, ElementRef, ViewChild, OnInit} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Router} from "@angular/router";
import {DomSanitizer} from '@angular/platform-browser';

import { NgModule }      from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule }   from '@angular/forms';
import { FormGroup, FormControl } from '@angular/forms';

import {UsersApiService} from '../users/users-api.service';
import {RecipesApiService} from "./recipes-api.service";

@Component({
  selector: 'recommender',
  templateUrl: './recommender.component.html',  
  styleUrls: ['../app.component.css']
})
export class RecommenderComponent implements OnInit{
 @ViewChild('uploadFile') uploadFile: ElementRef;

 constructor(private recipesApi: RecipesApiService, 
              private usersApi: UsersApiService, 
              private router: Router,
              private renderer: Renderer,
              private sanitizer: DomSanitizer) { }

  showLoading = false;
  sentImage = false;
  url = '';

  ingredients = ''
  recipesList = []


  ngOnInit(){
    if (!this.usersApi.loggedIn()){
      this.router.navigate(['/login'])
    }
  }

  showImageBrowseDlg() {
    this.uploadFile.nativeElement.click();
  }

  fileChange(event) {

    let file = event.target.files[0];
    console.log(file);

    if (event.target.files && event.target.files[0]) {
      var reader = new FileReader();
      reader.onload = (event: ProgressEvent) => {
        this.url = (<FileReader>event.target).result;
      }
      reader.readAsDataURL(event.target.files[0]);
    }

    let formData: FormData = new FormData();
    formData.append('image', file, file.name);

    console.log(formData);
    this.recipesApi
      .getSuggestions(formData)
      .subscribe(
        result => {
          this.ingredients = result['ingredients'];

          this.recipesList = result['recipes'];
          for (let recipe of this.recipesList){
            recipe['imgsrcsafe'] = this.sanitizer.bypassSecurityTrustStyle(`url(${recipe.imgsrc})`);
          }

          this.showLoading = false;
        },
        error => {
          console.log(error);
          this.showLoading = false;
          this.ingredients = "Unknown error occurred.";
        }
      );

      this.showLoading = true;
      this.sentImage = true;
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