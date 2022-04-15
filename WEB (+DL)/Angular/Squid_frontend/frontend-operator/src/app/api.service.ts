import { Injectable } from '@angular/core';
import { throwError } from 'rxjs';
import { map,retry,catchError } from 'rxjs/operators';
import { errorHandlerService } from "./error-handler.service";
import {HttpClient, HttpHeaders} from '@angular/common/http'

@Injectable()
export class apiService
{
    apiServerUrl: string = "http://158.177.22.22:5000";
    urlChosensList: string[] = [];
    ipChosensList: string[] = [];
    constructor(private http: HttpClient,private err:errorHandlerService){}

    setList(list:string[])
    {
        for (var elm of list) {
            if(elm[0]>='0' && elm[0]<='9' && elm.length>3 && elm[3]==='.')
            {
                this.ipChosensList.push(elm);
            }
            else
            {
                this.urlChosensList.push(elm);
            }
        }
    }

    postList()
    {
        const httpOptions = {
            headers: new HttpHeaders({
            'Content-Type':  'application/json'
            })
        };
        let request_url = this.apiServerUrl+'/addlist'
        
        let data = {'url_list':this.urlChosensList , 'ip_list':this.ipChosensList}
        return this.http.post(request_url, data, httpOptions)
        .pipe(
            map(
                (response) => {
                    this.err.clearError();
                    this.urlChosensList = [];
                    this.ipChosensList = [];
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
                    this.urlChosensList = [];
                    this.ipChosensList = [];
                }
                else
                {
                    error = error['error'];
                    this.err.setCode(error['code']);
                    this.err.setMessage(error['message']);
                    this.urlChosensList = [];
                    this.ipChosensList = [];
                }
                return throwError('Something went wrong with posting the provided list');
                }
            )
        );
    }

}