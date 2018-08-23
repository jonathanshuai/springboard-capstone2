import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import {Observable} from 'rxjs/Observable';
import 'rxjs/add/operator/catch';
import {API_URL} from '../env';
import {Recipe} from './recipe.model';

@Injectable()
export class RecipesApiService {

  constructor(private http: HttpClient) {
  }

  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }

  // GET list of public, future events
  getRecipes(): Observable<Recipe[]> {
    return this.http
      .get<Recipe[]>(`${API_URL}/recipes`)
      .catch(RecipesApiService._handleError);
  }

  saveRecipe(recipe: Recipe): Observable<any> {
    return this.http
      .post(`${API_URL}/save_recipe`, recipe);
  }

  deleteRecipe(recipe: Recipe): Observable<any> {
    return this.http
      .post(`${API_URL}/delete_recipe`, recipe);
  }

  getSuggestions(formData): Observable<Recipe[]> {
    return this.http
      .post<Recipe[]>(`${API_URL}/recommender`, formData);
  }

  sendDummy(): Observable<any> {
    return this.http
      .post(`${API_URL}/dummy`, '');
  }

}