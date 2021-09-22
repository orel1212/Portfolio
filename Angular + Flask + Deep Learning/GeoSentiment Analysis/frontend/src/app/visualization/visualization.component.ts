import { Component, OnInit } from '@angular/core';
import { tweetsService } from '../shared/tweets.service';
import { errorHandlerService } from '../shared/error-handler.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-visualization',
  templateUrl: './visualization.component.html',
  styleUrls: ['./visualization.component.css']
})
export class VisualizationComponent implements OnInit {

  isLoading: boolean = false; 
  isError: boolean = false;
  notFoundTweets:boolean=false;
  isSubcribeExecuted=false;
  barPercentage = 0;
  refreshIntervalId;
  numberOfTweets: number = 0;
  minDate: Date;
  maxDate: Date;
  collapse:boolean = false;
  errorMsg:string="";
  searchTerm:string="";
  constructor(private tweetsService: tweetsService,
    private errorHandler:errorHandlerService,
    private router: Router) { }

  ngOnInit() {
    this.isLoading = true;
    this.refreshIntervalId = setInterval(
      () => {
        let increase = 1;
        if (this.barPercentage + increase < 100)
          this.barPercentage += increase;
        else 
        { 
          this.barPercentage = 100;
          setTimeout(()=>{this.isLoading = false;},1000);
          clearInterval(this.refreshIntervalId);
        }
       }
       , 1800);

    this.tweetsService.tweetsReceived
    .subscribe((isSuccess: boolean) => 
    { 
      this.isSubcribeExecuted=true;
      if (isSuccess === true){
        this.barPercentage = 100;
        if(this.tweetsService.getTweets().length===0)
        {
          this.notFoundTweets=true;
        }
        else
        {
          this.notFoundTweets=false;
          this.numberOfTweets = this.tweetsService.getTweets().length;
          this.minDate = this.tweetsService.getMinDate();
          this.maxDate = this.tweetsService.getMaxDate();
          this.searchTerm= this.tweetsService.getLastSearchTerm();
        }
       this.errorMsg="";
      }
      else
      {
        let errorCode=this.errorHandler.getCode();
        if(errorCode===1)
        {
          this.errorMsg = "tweets are currently not avialable.";
        }
        else if(errorCode===9)
        {
          localStorage.clear();
          this.router.navigate(['/login']);
        }
        else
        {
          this.errorMsg = this.errorHandler.getMessage();
        }
      }

      this.isError = !isSuccess;
      this.isLoading = false;
    })
  }

  toggleCollapse() {
    this.collapse = !this.collapse;
  }

}
