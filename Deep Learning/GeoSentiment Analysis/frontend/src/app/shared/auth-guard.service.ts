import { apiService } from './api.service';
import { CanActivate, Router, ActivatedRouteSnapshot, RouterStateSnapshot, ActivatedRoute } from "@angular/router";
import { Observable } from "rxjs";
import jwt_decode from "jwt-decode";
import { Injectable } from "@angular/core";
@Injectable()
export class AuthGuard implements CanActivate {
    private refreshTokenInterval;
    constructor(private router:Router,
                private apiService:apiService,private route: ActivatedRoute){
                }

    
    canActivate(routesnap: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean | Observable<boolean> | Promise<boolean> 
    {
        this.route.queryParams
      .subscribe(
        (params) => {
            if(params['mode']==='demo')
            {
                return true;
            }
            else if(params['mode']==='online')
            {
                if (localStorage.getItem('refresh_token') === null)
                {
                    this.router.navigate(['/login']);
                    return false;
                }
                let refresh_token = localStorage.getItem('refresh_token');
                try
                {
                    let decoded = jwt_decode(refresh_token);
                    let exp_date:number = decoded['exp'];
                    let js_exp_date=new Date(exp_date * 1000);
                    let current_date=new Date();
                    if(current_date.getTime()-js_exp_date.getTime() >= 0)
                    {
                        this.router.navigate(['/login']);
                        return false;
                    }
                    else
                    {
                        return true;
                    }
                    
                }
                catch(e)
                {
                    this.router.navigate(['/login']);
                    return false;
                }
            }

        }
      );
      return true;
        
        
        
    }
}