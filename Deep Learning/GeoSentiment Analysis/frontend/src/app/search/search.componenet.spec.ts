import { of } from "rxjs";
import { Tweet } from "../shared/tweet.model";
import { SearchComponent } from "./search.component";
import { tweetsService } from "../shared/tweets.service";
import { TestBed } from "@angular/core/testing";
import { apiService } from "../shared/api.service";
import { ReactiveFormsModule } from "@angular/forms";
import { RouterTestingModule } from '@angular/router/testing';
import { errorHandlerService } from "../shared/error-handler.service";


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

 describe('Component: Search', () => {

      
    beforeEach(() => { 
        TestBed.configureTestingModule({
            declarations: [SearchComponent],
            providers: [tweetsService,
            {provide: apiService, useClass: MockApiService},
            errorHandlerService],
            imports:[ReactiveFormsModule,RouterTestingModule]
          });
        

    });
    
    it('check if trends updated in the trendingHashtags array', (done) => { 
        let fixture=TestBed.createComponent(SearchComponent);
        let component=fixture.componentInstance;
        let api_service=TestBed.get(apiService);
        api_service.getTrends();
        done(); 
        fixture.detectChanges();
        expect(component.trendingHashtags).toEqual(['trump','putin','eurovision']);
        });
    
    it('check if trends updated in the trends list', (done) => { 
        let fixture=TestBed.createComponent(SearchComponent);
        let component=fixture.componentInstance;
        let api_service=TestBed.get(apiService);
        api_service.getTrends();
        done(); 
        fixture.detectChanges();
        expect(fixture.debugElement.nativeElement.querySelector("#trends").innerText).toContain('trump');
        });
    it('check if selected trend "trump" give us tweets', (done) => { 
        let fixture=TestBed.createComponent(SearchComponent);
        let component=fixture.componentInstance;
        let api_service=TestBed.get(apiService);
        api_service.getTrends();
        component.getPredictionsBySelectedHashtag();
        let tweets_service=TestBed.get(tweetsService);
        let tweet:Tweet=tweets_service.getTweet(0);
        done(); 
        fixture.detectChanges();
        expect(tweet.sentiment).toEqual('positive');
        });
    });
  