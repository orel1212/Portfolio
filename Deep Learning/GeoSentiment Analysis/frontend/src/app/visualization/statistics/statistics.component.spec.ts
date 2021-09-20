import { of } from "rxjs";
import { Tweet } from "../../shared/tweet.model";
import { tweetsService } from "../../shared/tweets.service";
import { TestBed } from "@angular/core/testing";
import { apiService } from "../../shared/api.service";
import { ReactiveFormsModule } from "@angular/forms";
import { RouterTestingModule } from '@angular/router/testing';
import { StatisticsComponent } from "./statistics.component";
import { errorHandlerService } from "../../shared/error-handler.service";


 class MockApiService
 {
     trends={'trends':['trump','putin','eurovision']};
     tweets={'tweets':[{
        'text' : 'trump is great for america',
        'sentiment' : 'positive',
        'geolocation' : 'null',
        'country_prediction' : 'IL',
        'created_at' : "Thu Dec 06 15:24:15 +0000 2017",
        'tweet_lang' : "EN"}
        ,{
            'text' : 'trump is sux',
            'sentiment' : 'negative',
            'geolocation' : 'null',
            'country_prediction' : 'IL',
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

 describe('Component: Statistics', () => {

      
    beforeEach(() => { 
        TestBed.configureTestingModule({
            declarations: [StatisticsComponent],
            providers: [tweetsService,
            {provide: apiService, useClass: MockApiService},
                        errorHandlerService],
            imports:[ReactiveFormsModule,RouterTestingModule]
          });
        

    });
    
    it('check if number of countries in the statistics table equals the one in the tweetsService', (done) => { 
        let api_service=TestBed.get(apiService);
        let tweets_service=TestBed.get(tweetsService);
        tweets_service.getPredictions("trump");
        api_service.getTweets();
        done();
        let fixture=TestBed.createComponent(StatisticsComponent);
        let component=fixture.componentInstance;
        fixture.detectChanges();
        expect(component.countryStats.length).toEqual(Object.keys(tweets_service.getCountries()).length);
        });
    it('check if country Israel is in the statistics html table', (done) => { 
        let api_service=TestBed.get(apiService);
        let tweets_service=TestBed.get(tweetsService);
        tweets_service.getPredictions("trump");
        api_service.getTweets();
        done();
        let fixture=TestBed.createComponent(StatisticsComponent);
        let component=fixture.componentInstance;
        fixture.detectChanges();
        expect(fixture.debugElement.nativeElement.querySelector("tbody tr td").innerText).toContain('Israel');
        });
    
    });
  