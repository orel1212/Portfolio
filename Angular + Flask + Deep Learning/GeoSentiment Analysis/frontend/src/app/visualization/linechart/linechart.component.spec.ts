import { of } from "rxjs";
import { Tweet } from "../../shared/tweet.model";
import { tweetsService } from "../../shared/tweets.service";
import { apiService } from "../../shared/api.service";
import { TestBed } from "@angular/core/testing";
import { ReactiveFormsModule } from "@angular/forms";
import { RouterTestingModule } from '@angular/router/testing';
import { LinechartComponent } from "./linechart.component";
import { Ng2GoogleChartsModule } from "ng2-google-charts";
import { errorHandlerService } from "../../shared/error-handler.service";



 class MockApiService
 {
     trends={'trends':['trump','putin','eurovision']};
     tweets={'tweets':[{
        'text' : 'trump is great for america',
        'sentiment' : 'positive',
        'geolocation' : 'null',
        'country_prediction' : 'US',
        'created_at' : "Thu Dec 06 15:24:15 +0000 2017",
        'tweet_lang' : "EN"}
        ,{
            'text' : 'trump is sux',
            'sentiment' : 'negative',
            'geolocation' : 'null',
            'country_prediction' : 'US',
            'created_at' : "Thu Dec 06 15:24:15 +0000 2017",
            'tweet_lang' : "HE"}
       
     ]};
     getTrends(){
        return of(this.trends);
    }
    getDemoTrends(){
        return of(this.trends);
    }


    getTweets(text: string){
        return of(this.tweets);
    }
    getDemoTweets(text: string){
        return of(this.tweets);
    }
 }

 describe('Component: Linechart', () => {

      
    beforeEach(() => { 
        TestBed.configureTestingModule({
            declarations: [LinechartComponent],
            providers: [tweetsService,
            {provide: apiService, useClass: MockApiService},
            errorHandlerService],
            imports:[ReactiveFormsModule,RouterTestingModule,Ng2GoogleChartsModule]
          });

    });
    
    it('check that the date in the line chart is 6/12/2017 only ', (done) => { 
        
        let api_service=TestBed.get(apiService);
        let tweets_service=TestBed.get(tweetsService);
        tweets_service.getPredictions("trump");
        api_service.getTweets();
        done();
        let fixture=TestBed.createComponent(LinechartComponent);
        let component=fixture.componentInstance;
        fixture.detectChanges();
        let linechart_data=component.lineChartData['dataTable'];
        expect(linechart_data.length-1).toEqual(1);//-1 because of the heading, 1 stands for 6/12/2017
        expect(linechart_data[1][0]).toEqual('6/12/2017');//linechart_data[1][0] is the date itself in the data
        });
    });
  