import { apiService } from './api.service';
import { Injectable, OnInit } from "@angular/core";
import { Tweet } from "./tweet.model";
import { Subject } from "rxjs";

import { Country } from './country.model';


@Injectable()
export class tweetsService{

    private tweets: Tweet[] = [
    ];
    private minDate: Date;
    private maxDate: Date;
    
    private countries: {[name: string] : Country } = {};
    private words: {[name:string] : number}={};
    private hashtags: {[name:string] : number}={};
    private searchTerm:string="";
    tweetsReceived = new Subject();
    constructor(private apiService: apiService) { 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "EN"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "EN"));      
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "EN"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "EN"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "EN"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "EN"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "EN"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "EN")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "HE")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "HE"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "HE"));    
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "HE")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "HE"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "HE")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "HE")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "HE"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "HE")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "HE")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "HE"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "HE")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:25:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','GB', new Date("Thu Dec 03 15:24:15 +0000 2017"), "AR")); 
      // this.addTweet(new Tweet('positive','US', new Date("Thu Dec 06 15:24:15 +0000 2017"), "AR")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:24:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:25:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','ZA', new Date("Thu Dec 03 15:24:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:25:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','ZA', new Date("Thu Dec 03 15:24:15 +0000 2017"), "AR")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:25:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','ZA', new Date("Thu Dec 03 15:24:15 +0000 2017"), "AR")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:25:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','ZA', new Date("Thu Dec 03 15:24:15 +0000 2017"), "AR")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:25:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','ZA', new Date("Thu Dec 03 15:24:15 +0000 2017"), "AR")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:25:15 +0000 2017"), "AR"));
      // this.addTweet(new Tweet('positive','RU', new Date("Thu Dec 03 15:24:15 +0000 2017"), "AR")); 
      // this.addTweet(new Tweet('negative','IL', new Date("Thu Nov 09 15:25:15 +0000 2017"), "RU"));
      // this.addTweet(new Tweet('positive','RU', new Date("Thu Dec 03 15:24:15 +0000 2017"), "RU"));
    }

    getTweets() {
      return this.tweets;
    }
  
    getTweet(index: number) {
      return this.tweets[index];
    }

    addTweet(tweet: Tweet) {

      if (this.tweets.length === 0) {
        this.minDate = tweet.createdAt;
        this.maxDate = tweet.createdAt;
      }
      else {
        if (tweet.createdAt < this.minDate)
          this.minDate = tweet.createdAt;
        else if (tweet.createdAt > this.maxDate)
          this.maxDate = tweet.createdAt;
      }
      let country = tweet.location;
      country = country.toLowerCase();
      if (this.countries[country] == undefined) {
        this.countries[country] = new Country(country);
      }
      if (tweet.sentiment === 'positive') {
        this.countries[country].incPositive();
      }
      else {
        this.countries[country].incNegative();
      }
      this.countries[country].addLanguage(tweet.lang);
      this.tweets.push(tweet);
    }

    getCountries() {
      return this.countries;
    }
    getWords()
    {
      return this.words;
    }
    getHashtags()
    {
      return this.hashtags;
    }
    getMinDate() {
      return this.minDate;
    }

    getMaxDate() {
      return this.maxDate;
    }

    updateWordsAndHashtags(text:string)
    {
      text = text.replace(/(https?:\/\/[^\s]+)/g, " ");
      text= text.replace(/[\n\s\r\t.,!?%$!~:;'"\/><^&*-]+/g, " ");
      text= text.replace(/@@+/g, " ");
      let splitted_text = text.split(" ");
      for (let text_element of splitted_text){
        text_element = text_element.toLowerCase();
        if( text_element!=="")
        {
          if (text_element[0]==="#")
          {
            if(this.hashtags[text_element] === undefined)
            {
              this.hashtags[text_element] = 1;
            }
            else
            {
              this.hashtags[text_element] += 1;
            } 
          }
          else if(text_element.length >1 || text_element==="i" || text_element==="a" || text_element==="v" || text_element==="o")
          {
            if(this.words[text_element] === undefined)
            {
              this.words[text_element] = 1;
            }
            else
            {
              this.words[text_element] += 1;
            } 
          }
        }
      }
    }
  clearObjects()
  {
    this.tweets = [];
    this.countries = {};
    this.words={};
    this.hashtags={};
  }
  getPredictions(searchInput: string){
    this.clearObjects();
    let isSuccess: boolean = false;
    this.searchTerm=searchInput;
    this.apiService.getTweets(searchInput)
    .subscribe(
      (tweets) => {
        
        for (let tweet of tweets['tweets']){
          this.updateWordsAndHashtags(tweet['text']);
          this.addTweet(new Tweet(tweet['sentiment'],tweet['country_prediction'],  new Date(tweet['created_at']), tweet['tweet_lang']));
        }
        isSuccess = true;
        
        this.tweetsReceived.next(isSuccess);
      },
      (error) => this.tweetsReceived.next(isSuccess)
    );
  }

  getDemoPredictions(searchInput: string){
    this.clearObjects();
    let isSuccess: boolean = false;
    this.searchTerm=searchInput;
    this.apiService.getDemoTweets(searchInput)
    .subscribe(
      (tweets) => {
        for (let tweet of tweets['tweets']){
          this.updateWordsAndHashtags(tweet['text']);
          this.addTweet(new Tweet(tweet['sentiment'],tweet['country_prediction'],  new Date(tweet['created_at']), tweet['tweet_lang']));
        }
        isSuccess = true;
        this.tweetsReceived.next(isSuccess);
      },
      (error) => this.tweetsReceived.next(isSuccess)
    );
  }

  getLastSearchTerm()
  {
    return this.searchTerm;
  }


}
