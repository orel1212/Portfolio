import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router, ActivatedRoute } from '@angular/router';
import { apiService } from '../api.service';
import { errorHandlerService } from '../error-handler.service';

@Component({
  selector: 'app-status',
  templateUrl: './status.component.html',
  styleUrls: ['./status.component.css']
})
export class StatusComponent implements OnInit,OnDestroy {

  isArrived:boolean = false;
  barPercentage = 0;
  refreshIntervalId;
  errorMsg:string="";
  isCodeError:boolean = false;

  constructor(
    private router: Router,
    private apiService: apiService,
    private errorHandler:errorHandlerService
  ) { }

  ngOnInit(): void {
    this.isArrived = false;
    this.refreshIntervalId = setInterval(
      () => {
        let increase = 2;
        if (this.barPercentage + increase < 100)
          this.barPercentage += increase;
        else 
        { 
          this.barPercentage = 100;
          setTimeout(()=>{this.isArrived = true;},1000);
          clearInterval(this.refreshIntervalId);
        }
       }
       , 200);
       this.apiService.postList()
       .subscribe(
         (response) => {
           this.isArrived = true;
           this.isCodeError = false;
            /*
            to provide few seconds to see the message after the request!
            */
            setTimeout(()=>{this.router.navigate(['/']);},5000);
              
         },
         (error) => { 
           this.isArrived = true;
           this.errorMsg = this.errorHandler.getMessage();
           this.isCodeError = true;
                  /*
          to provide few seconds to see the message after the request!
          */
         setTimeout(()=>{this.router.navigate(['/']);},5000);
         }
       );
      
  }

  ngOnDestroy(): void {
    
  }

}
