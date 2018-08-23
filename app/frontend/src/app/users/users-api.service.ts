import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import {Observable} from 'rxjs/Observable';
import 'rxjs/add/operator/catch';
import {API_URL} from '../env';
import {User} from './user.model';
import {shareReplay, tap } from 'rxjs/operators';

@Injectable()
export class UsersApiService {

  constructor(private http: HttpClient) {
  }

  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }

  registerUser(user: User): Observable<any> {
    return this.http
      .post(`${API_URL}/register`, user);
  }

  authenticateUser(user: User): Observable<any> {
    return this.http
      .post(`${API_URL}/authenticate`, user).pipe(
        tap(res => this.setSession(res)), 
        shareReplay()
      );
  }

  setSession(authResult) {
    localStorage.setItem('id_token', authResult.auth_token);
  }

  loggedIn(){
    if (localStorage.getItem('id_token')){
      return true;
    }
    return false;
  } 

  logout() {
    localStorage.removeItem("id_token");
  }
}