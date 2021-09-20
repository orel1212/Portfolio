using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace Project.ModelBinders
{
    public class CustomerBinder : IModelBinder
    {

        public object BindModel(ControllerContext controllerContext, ModelBindingContext bindingContext)
        {
            HttpContextBase objContext = controllerContext.HttpContext;
            string cid = objContext.Request.Form["id"].ToString();
            string cname = objContext.Request.Form["name"].ToString();
            string pass = objContext.Request.Form["password"].ToString();
            Customer cust = new Customer(cid, cname, pass);
            return cust;
        }
    }
}