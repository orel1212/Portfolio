using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace Project.ModelBinders
{
    public class WorkerBinder : IModelBinder
    {

        public object BindModel(ControllerContext controllerContext, ModelBindingContext bindingContext)
        {
            HttpContextBase objContext = controllerContext.HttpContext;
            string wid = objContext.Request.Form["id"].ToString();
            string wname = objContext.Request.Form["name"].ToString();
            string pass = objContext.Request.Form["password"].ToString();
            string typeofworker = objContext.Request.Form["worker"].ToString();
            Worker worker = null;
            if (typeofworker.Equals("cashier"))
                worker = new Cashier(wid, wname, pass);
            else
                worker = new Manager(wid, wname, pass);
            return worker;
        }
    }
}