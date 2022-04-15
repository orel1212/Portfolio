import { apiService } from './shared/api.service';
import { Component } from '@angular/core';
import jwt_decode from "jwt-decode";
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'GeoSentiment Analysis';
  private refreshTokenInterval;
  constructor(private apiService:apiService){
        if( localStorage.getItem('refresh_token')!==null)
        {
            this.refreshTokenFunc();
        }
        this.refreshTokenInterval = setInterval(this.refreshTokenFunc.bind(this),300000);
  }

  refreshTokenFunc(){
    let access_token = localStorage.getItem('access_token');
    try
    {
        let decoded = jwt_decode(access_token);
        let exp_date:number = decoded['exp'];
        let js_exp_date=new Date(exp_date * 1000);
        let current_date=new Date();
        let interval = current_date.getTime()-js_exp_date.getTime();
        if(interval >= 0  ||  interval <= -300000)
        {
            this.apiService.refreshAccessToken()
            .subscribe(
                (response) => {
                },
                (error) => 
                {
                }
            );
        }
        
    }
    catch(e)
    {
        this.apiService.refreshAccessToken()
        .subscribe(
            (response) => {
            },
            (error) => 
            {
            }
        );
    }
  }



  
}
