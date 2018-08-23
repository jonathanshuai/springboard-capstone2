import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import {Observable} from 'rxjs/Observable';
import 'rxjs/add/operator/catch';
import {API_URL} from '../env';
import {Restriction} from './restriction.model';

@Injectable()
export class RestrictionsApiService {

  constructor(private http: HttpClient) {
  }

  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }

  // GET list of public, future events
  getRestrictions(): Observable<Restriction> {
    return this.http
      .get<Restriction>(`${API_URL}/restrictions`)
      .catch(RestrictionsApiService._handleError);
  }

  saveRestrictions(restriction: Restriction): Observable<any> {
    return this.http
      .post(`${API_URL}/update_restriction`, restriction);
  }
}