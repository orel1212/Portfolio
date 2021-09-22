import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { ReactiveFormsModule } from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ErrorPageComponent } from './error-page/error-page.component';
import { HomeComponent } from './home/home.component';
import { HeaderComponent } from './header/header.component';
import { AboutComponent } from './about/about.component';
import { AddlistComponent } from './addlist/addlist.component';
import { HttpClientModule } from '@angular/common/http';
import { apiService } from './api.service';
import { errorHandlerService } from './error-handler.service';
import { StatusComponent } from './status/status.component';

@NgModule({
  declarations: [
    AppComponent,
    ErrorPageComponent,
    HomeComponent,
    HeaderComponent,
    AboutComponent,
    AddlistComponent,
    StatusComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    ReactiveFormsModule,
    HttpClientModule
  ],
  providers: [apiService,errorHandlerService],
  bootstrap: [AppComponent]
})
export class AppModule { }
