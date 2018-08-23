import { Injectable } from '@angular/core';
import {HttpRequest, 
        HttpHandler, 
        HttpEvent, 
        HttpInterceptor, 
        HttpResponse, 
        HttpErrorResponse} from '@angular/common/http';

import { Observable } from 'rxjs/Observable';
import 'rxjs/add/operator/do';

@Injectable()
export class TokenInterceptor implements HttpInterceptor {
  constructor() {}

  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    request = request.clone({
      setHeaders: {
        Authorization: `Bearer ${localStorage.getItem("id_token")}`
      }
    });

  //   return next.handle(request);
  // }

  return next.handle(request).do((event: HttpEvent<any>) => {
    if (event instanceof HttpResponse) {
        // do stuff with response if you want
        console.log(event)
      }
    }, 
    (err: any) => {
      console.log(err)
      if (err instanceof HttpErrorResponse) {
        if (err.status === 401) {
          // redirect to the login route
          // or show a modal
        }
      }
    });
  }

}