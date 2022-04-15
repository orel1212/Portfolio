import { Injectable } from "@angular/core";
import { Headers, Http, Response } from '@angular/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map, retry, mergeMap } from 'rxjs/operators'
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { errorHandlerService } from "./error-handler.service";

@Injectable()
export class apiService {
    apiServerUrl: string = "https://timorsatarov.me";
    constructor(private http: HttpClient,private err:errorHandlerService){}



    getTrends(){

        let jwt_access_token = localStorage.getItem('access_token');
        
        const httpOptions = {
            headers: new HttpHeaders({
              'Authorization':  'Bearer ' + jwt_access_token
            })
          };

        
        return this.http.get(this.apiServerUrl + "/api/trends", httpOptions)
            .pipe(
                map(
                    (response) => {
                        this.err.clearError();
                        return response;
                    }
                )
                ,retry(1)
                ,catchError(
                    (error: Response) => {
                    if(error['status']!==undefined && error['status']===0)
                    {
                        this.err.setCode(1);
                        this.err.setMessage(error['message']);
                    }
                    else
                    {
                        error = error['error'];
                        this.err.setCode(error['code']);
                        this.err.setMessage(error['message']);
                    }
                    return throwError('Something went wrong with the TRENDING API');
                    }
                )
            );
    }


    getTweets(text: string){
        let jwt_access_token = localStorage.getItem('access_token');
        
        const httpOptions = {
            headers: new HttpHeaders({
              'Authorization':  'Bearer ' + jwt_access_token
            })
          };

        let request_url = this.apiServerUrl + "/api/predictions/" +  encodeURIComponent(text);
        return this.http.get(request_url, httpOptions)
            .pipe(
                map(
                    (response) => {
                        if(response['tweets'].length===0)
                        {
                            this.err.setCode(13);
                            this.err.setMessage("tweets not found");
                        }
                        else
                        {
                            this.err.clearError();
                        }
                        return response;
                    }
                )
                ,catchError(
                    (error: Response) => {
                    if(error['status']!==undefined && error['status']===0)
                    {
                        this.err.setCode(1);
                        this.err.setMessage(error['message']);
                    }
                    else
                    {
                        error = error['error'];
                        this.err.setCode(error['code']);
                        this.err.setMessage(error['message']);
                    }
                    return throwError('Something went wrong with the PREDICTIONS TEXT');

                    }
                )
            );
    }

    getDemoTrends(){
        return this.http.get(this.apiServerUrl + "/api/demo/trends")
            .pipe(
                map(
                    (response) => {
                        this.err.clearError();
                        return response;
                    }
                )
                ,retry(1)
                ,catchError(
                    (error: Response) => {
                    if(error['status']!==undefined && error['status']===0)
                    {
                        this.err.setCode(1);
                        this.err.setMessage(error['message']);
                    }
                    else
                    {
                        error = error['error'];
                        this.err.setCode(error['code']);
                        this.err.setMessage(error['message']);
                    }
                    return throwError('Something went wrong with the DEMO TRENDING API');
                    }
                )
            );        
    }

    getDemoTweets(text: string){
        let request_url = this.apiServerUrl + "/api/demo/predictions/" +  encodeURIComponent(text);
        return this.http.get(request_url)
            .pipe(
                map(
                    (response) => {
                        if(response['tweets'].length===0)
                        {
                            this.err.setCode(13);
                            this.err.setMessage("tweets not found");
                        }
                        else
                        {
                            this.err.clearError();
                        }
                        return response;
                    }
                )
                ,retry(1)
                ,catchError(
                    (error: Response) => {
                    if(error['status']!==undefined && error['status']===0)
                    {
                        this.err.setCode(1);
                        this.err.setMessage(error['message']);
                    }
                    else
                    {
                        error = error['error'];
                        this.err.setCode(error['code']);
                        this.err.setMessage(error['message']);
                    }
                    return throwError('Something went wrong with the DEMO PREDICTIONS TEXT');
                    }
                )
            );
    }


    requestTwitterAuthUrl() {
        let request_url = this.apiServerUrl + "/api/twitter_auth"
        return this.http.get(request_url)
            .pipe(
                map(
                    (response) => {
                        this.err.clearError();
                        return response;
                    }
                )
                ,retry(1)
                ,catchError(
                    (error: Response) => {
                    if(error['status']!==undefined && error['status']===0)
                    {
                        this.err.setCode(1);
                        this.err.setMessage(error['message']);
                    }
                    else
                    {
                        error = error['error'];
                        this.err.setCode(error['code']);
                        this.err.setMessage(error['message']);
                    }
                    return throwError('Something went wrong with the requestTwitterAuthUrl');
                    }
                )
            );
    }

    postTwitterVerifierCode(tokenJson: string) {

        const httpOptions = {
            headers: new HttpHeaders({
              'Content-Type':  'application/json'
            })
          };
        let request_url = this.apiServerUrl + "/api/twitter_verify_auth"
        return this.http.post(request_url, tokenJson, httpOptions)
            .pipe(
                map(
                    (response) => {
                        localStorage.setItem('access_token', response['access_token']);
                        localStorage.setItem('refresh_token', response['refresh_token']);
                        this.err.clearError();
                        return response;
                    }
                )
                ,retry(1)
                ,catchError(
                    (error: Response) => {
                    if(error['status']!==undefined && error['status']===0)
                    {
                        this.err.setCode(1);
                        this.err.setMessage(error['message']);
                    }
                    else
                    {
                        error = error['error'];
                        this.err.setCode(error['code']);
                        this.err.setMessage(error['message']);
                    }
                    return throwError('Something went wrong with the postTwitterVerifierCode');
                    }
                )
            );
    }

    refreshAccessToken() {
        let jwt_refresh_token = localStorage.getItem('refresh_token');
        const httpOptions = {
            headers: new HttpHeaders({
              'Authorization':  'Bearer ' + jwt_refresh_token
            })
          };
        let request_url = this.apiServerUrl + "/api/refresh_token"


        return this.http.get(request_url, httpOptions)
            .pipe(
                map(
                    (response) => {
                    localStorage.setItem('access_token', response['access_token']);
                    return response;
                    }
                )
                ,retry(1)
                ,catchError(
                    (error: Response) => {
                    if(error['status']!==undefined && error['status']===0)
                    {
                        this.err.setCode(1);
                        this.err.setMessage(error['message']);
                    }
                    else
                    {
                        error = error['error'];
                        this.err.setCode(error['code']);
                        this.err.setMessage(error['message']);
                    }
                    return throwError('Something went wrong with the refreshAccessToken');
                    }
                )
            );
    }
    
}