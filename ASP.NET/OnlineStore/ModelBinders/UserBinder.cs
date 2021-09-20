
using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace Project.ModelBinders
{
    public class UserBinder : IModelBinder
    {

        public object BindModel(ControllerContext controllerContext, ModelBindingContext bindingContext)
        {
            HttpContextBase objContext = controllerContext.HttpContext;
            string id = objContext.Request.Form["id"].ToString();
            string pass = objContext.Request.Form["password"].ToString();
            User user = new User(id, pass);
            return user;
        }
    }
}