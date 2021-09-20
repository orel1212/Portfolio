
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

import { AppComponent } from './app.component';
import { HeaderComponent } from './header/header.component';
import { HomeComponent } from './home/home.component';
import { SearchComponent } from './search/search.component';
import { AppRoutingModule } from './app-routing.module';
import { ErrorPageComponent } from './error-page/error-page.component';
import { GeochartComponent } from './visualization/geochart/geochart.component';
import { Ng2GoogleChartsModule } from 'ng2-google-charts';
import { tweetsService } from './shared/tweets.service';
import { StatisticsComponent } from './visualization/statistics/statistics.component';
import { apiService } from './shared/api.service';
import { VisualizationComponent } from './visualization/visualization.component';
import { AboutComponent } from './about/about.component';
import { WordCloudComponent } from './visualization/word-cloud/word-cloud.component';
import { TagCloudModule } from 'angular-tag-cloud-module';
import { LinechartComponent } from './visualization/linechart/linechart.component';
import { ApiComponent } from './api/api.component';
import { ApiInnerTemplateComponent } from './api/api-inner-template/api-inner-template.component';
import { LoginComponent } from './login/login.component';
import { AuthGuard } from './shared/auth-guard.service';
import { errorHandlerService } from './shared/error-handler.service';
import { languageConverter } from './shared/langConverter.pipe';
import { countryConverter } from './shared/countryConverter.pipe';
import { LangCloudComponent } from './visualization/lang-cloud/lang-cloud.component';
import { HashtagCloudComponent } from './visualization/hashtag-cloud/hashtag-cloud.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    HomeComponent,
    SearchComponent,
    ErrorPageComponent,
    GeochartComponent,
    StatisticsComponent,
    VisualizationComponent,
    AboutComponent,
    WordCloudComponent,
    LinechartComponent,
    ApiComponent,
    ApiInnerTemplateComponent,
    LoginComponent,
    countryConverter,
    languageConverter,
    LangCloudComponent,
    HashtagCloudComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    Ng2GoogleChartsModule,
    ReactiveFormsModule,
    HttpClientModule,
    TagCloudModule
  ],
  providers: [tweetsService, apiService,AuthGuard,errorHandlerService],
  bootstrap: [AppComponent]
})
export class AppModule { }
