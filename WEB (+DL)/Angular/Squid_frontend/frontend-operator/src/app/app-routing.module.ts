import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { AboutComponent } from './about/about.component';
import { ErrorPageComponent } from './error-page/error-page.component';
import { AddlistComponent } from './addlist/addlist.component';
import { StatusComponent } from './status/status.component';


const routes: Routes = [
  { path: '', component: HomeComponent, pathMatch: 'full' },
  { path: 'home', component: HomeComponent},
  { path: 'addlist', component: AddlistComponent},
  { path: 'about', component: AboutComponent},
  { path: 'status', component: StatusComponent},
  { path: 'not-found', component: ErrorPageComponent, data: {message: 'Sorry,the page cannot be found!',additionalInfo:'You may have typed the URL incorrectly or used outdated link!'} },
  { path: '**', redirectTo: '/not-found' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
