import { tweetsService } from './../shared/tweets.service';
import { apiService } from './../shared/api.service';
import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { FormGroup, FormControl, FormArray, Validators } from '@angular/forms';
import { errorHandlerService } from '../shared/error-handler.service';

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit {
  mode:string='online';

  trendingHashtags: string[] = [];
  isLoadingTrends: boolean = true;
  selectedIndex: number = -1;
  searchForm: FormGroup;

  isTrendsAvailable: boolean = true;
  isError: boolean = false;
  errorMsg:string="";
  constructor(private route: ActivatedRoute,
              private apiService: apiService,
              private tweetsService: tweetsService,
              private router: Router,
              private errorHandler:errorHandlerService) { }

  ngOnInit() {
    if(this.route.snapshot.queryParams['mode'] !== undefined)
    {
      this.mode = this.route.snapshot.queryParams['mode'];
    }
    else
    {
      this.initTrends();
      this.initForm();
      this.selectedIndex=0;
    }
    this.route.queryParams
      .subscribe(
        (params) => {
          this.isLoadingTrends = true;
          this.isTrendsAvailable = true;
          this.isError = false;
          this.mode = params['mode'];
          if (this.mode === "online"){
            this.initTrends()
          }
          else{
            this.initDemoTrends();
          }
          this.initForm();

        }
      );

  }

  initDemoTrends() {
    this.apiService.getDemoTrends()
    .subscribe(
      (trends) => 
        {
          this.trendingHashtags = trends['trends'];
          this.isLoadingTrends = false;
          this.errorMsg="";
        },
      (error) => 
      {
        this.isLoadingTrends = false;
        this.isTrendsAvailable = false;
        let errorCode=this.errorHandler.getCode();
        if(errorCode===1)
        {
          this.errorMsg = "Trending hashtags are currently not avialable.";
        }
        else
        {
          this.errorMsg = this.errorHandler.getMessage();
        }
      }
    );
  }

  initTrends() {
    this.apiService.getTrends()
    .subscribe(
      (trends) => 
        {
          this.trendingHashtags = trends['trends'];
          this.isLoadingTrends = false;
          this.errorMsg="";
        },
      (error) => 
      {
        this.isLoadingTrends = false;
        this.isTrendsAvailable = false;
        let errorCode=this.errorHandler.getCode();
        if(errorCode===1)
        {
          this.errorMsg = "Trending hashtags are currently not avialable.";
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
    );
  }

  onSubmit() {
    if (this.searchForm.valid){
      let searchInput = this.searchForm.get("searchInput").value;
      if(this.mode === "online") {
        this.tweetsService.getPredictions(searchInput);
      }
      else {
        this.tweetsService.getDemoPredictions(searchInput);
      }
      this.router.navigate(['visualization']);
    }
  }

  onClick(i:number){
    this.selectedIndex = i;
  }

  isSelected(i:number){
    return this.selectedIndex===i;
  }

  private initForm() {
    let searchInput = '';
    let searchFormObject={};
    let validatorsArray=[Validators.required];
    if(this.mode==='demo')
    {
      validatorsArray.push(Validators.pattern("(#[A-Za-z][A-Za-z0-9]*|#[A-Za-z][A-Za-z0-9]* OR #[A-Za-z][A-Za-z0-9]*)+$"));
    }
    else if(this.mode==='online')
    {
      //
    }
    searchFormObject['searchInput']= new FormControl(searchInput, validatorsArray);
    this.searchForm = new FormGroup(searchFormObject);
  }


  getPredictionsBySelectedHashtag(){
    if(this.mode === "online") {
      this.tweetsService.getPredictions(this.trendingHashtags[this.selectedIndex]);
    }
    else {
      this.tweetsService.getDemoPredictions(this.trendingHashtags[this.selectedIndex]);
    }
  }
  
  
  

}
