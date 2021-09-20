
import { NgModule, Component } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { ErrorPageComponent } from './error-page/error-page.component';
import { HomeComponent } from './home/home.component';
import { SearchComponent } from './search/search.component';
import { GeochartComponent } from './visualization/geochart/geochart.component';
import { StatisticsComponent } from './visualization/statistics/statistics.component';
import { VisualizationComponent } from './visualization/visualization.component';
import { WordCloudComponent } from './visualization/word-cloud/word-cloud.component';
import { AboutComponent } from './about/about.component';
import { LinechartComponent } from './visualization/linechart/linechart.component';
import { ApiComponent } from './api/api.component';
import { LoginComponent } from './login/login.component';
import { AuthGuard } from './shared/auth-guard.service';
import { LangCloudComponent } from './visualization/lang-cloud/lang-cloud.component';
import { HashtagCloudComponent } from './visualization/hashtag-cloud/hashtag-cloud.component';

const appRoutes: Routes = [
  { path: '', component: HomeComponent, pathMatch: 'full' },
  { path: 'search',canActivate:[AuthGuard], component: SearchComponent},
  { path: 'visualization', component: VisualizationComponent, 
          children: 
          [
            { path: 'map', component: GeochartComponent},
            { path: 'statistics', component: StatisticsComponent},
            { path: 'word-cloud', component: WordCloudComponent},
            { path: 'hashtag-cloud', component: HashtagCloudComponent},
            { path: 'lang-cloud', component: LangCloudComponent},
            { path: 'date-graph', component: LinechartComponent}
          ]
  },
  { path: 'map', component: GeochartComponent},
  { path: 'about', component: AboutComponent},
  { path: 'api', component: ApiComponent},
  { path: 'login', component: LoginComponent},
  { path: 'not-found', component: ErrorPageComponent, data: {message: 'Sorry,the page cannot be found!',additionalInfo:'You may have typed the URL incorrectly or used outdated link!'} },
  { path: '**', redirectTo: '/not-found' }
];

@NgModule({
  imports: [
    RouterModule.forRoot(appRoutes)
  ],
  exports: [RouterModule]
})
export class AppRoutingModule {

}