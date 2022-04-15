using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.Web.Routing;

namespace Project
{
    public class RouteConfig
    {
        public static void RegisterRoutes(RouteCollection routes)
        {
            routes.IgnoreRoute("{resource}.axd/{*pathInfo}");
            routes.MapRoute(
                name: "CustomerSubmit",
                url: "Customer/Submit",
                defaults: new { controller = "Customer", action = "Submit", id = UrlParameter.Optional }
            );
            routes.MapRoute(
                name: "CustomerHome",
                url: "Customer/CustomerHome",
                defaults: new { controller = "Customer", action = "CustomerHome", id = UrlParameter.Optional }
            );
            routes.MapRoute(
                name: "ManagerHome",
                url: "Manager/ManagerHome",
                defaults: new { controller = "Manager", action = "ManagerHome", id = UrlParameter.Optional }
            );
            routes.MapRoute(
                name: "CashierHome",
                url: "Cashier/CashierHome",
                defaults: new { controller = "Cashier", action = "CashierHome", id = UrlParameter.Optional }
            );
            routes.MapRoute(
                name: "Authenticate",
                url: "User/Authenticate",
                defaults: new { controller = "User", action = "Authenticate", id = UrlParameter.Optional }
            );
            routes.MapRoute(
                name: "EditInfo",
                url: "User/EditInfo",
                defaults: new { controller = "User", action = "EditInfo", id = UrlParameter.Optional }
            );

            routes.MapRoute(
                name: "Direction",
                url: "{controller}/{action}/{id}",
                defaults: new { controller = "{controller}", action = "{action}", id = UrlParameter.Optional }
            );
            routes.MapRoute(
                name: "Default",
                url: "",
                defaults: new { controller = "Home", action = "Home", id = UrlParameter.Optional }
            );
        }
    }
}
