using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace Project.ModelBinders
{
    public class ProductBinder : IModelBinder
    {

        public object BindModel(ControllerContext controllerContext, ModelBindingContext bindingContext)
        {
            HttpContextBase objContext = controllerContext.HttpContext;
            decimal price;
            int amount;
            string pname = objContext.Request.Form["name"].ToString();
            try
            {
                 price = Convert.ToDecimal(objContext.Request.Form["price"].ToString());
                 amount = Convert.ToInt32(objContext.Request.Form["amount"].ToString());
            }
            catch (Exception e)
            {
                price = -1;
                amount = -1;
            }
            Product p = new Product(pname, price, amount);
            return p;
        }
    }
}