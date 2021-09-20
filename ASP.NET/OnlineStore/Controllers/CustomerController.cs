using Project.Models;
using Project.ModelBinders;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using Project.DAL;
using Project.ViewModels;
using System.Web.Security;

namespace Project.Controllers
{
    [Authorize(Roles="Customer")]
    public class CustomerController : Controller
    {
        public ActionResult CustomerHome()
        {
            return View();
        }
        [HttpGet]
        public ActionResult Order()
        {
            return View("Order", new ProductsVM());
        }
        [HttpPost]
        public ActionResult Order(Dictionary<string,int> cart,decimal total)
        {
            string custid = getCookieID();
            Order order = new Order(custid, decimal.Round(total, 2, MidpointRounding.AwayFromZero));
            DataLayer dal = new DataLayer();
            dal.Orders.Add(order);//in memory ,without commit to the db
            
            foreach (var PidPricePair in cart)
            {
                OrderDescription od = new OrderDescription(order.OID, Convert.ToInt16(PidPricePair.Key), PidPricePair.Value);
                dal.ODs.Add(od);//in memory ,without commit to the db
            }
            dal.SaveChanges();
            return RedirectToAction("CustomerHome");
        }

        [HttpGet]
        public ActionResult DeleteOrder()
        {
            return View("DeleteOrder",new OrderVM(getCookieID()));
        }

        [HttpPost]
        public ActionResult DeleteOrder(int oid)
        {
            DataLayer dal = new DataLayer();
            dal.ODs.RemoveRange(dal.ODs.Where(x => x.OID == oid));
            dal.Orders.RemoveRange(dal.Orders.Where(x => x.OID == oid));
            dal.SaveChanges();
            return RedirectToAction("CustomerHome");
        }
        [HttpGet]
        public ActionResult GetOpennedOrdersByJson()
        {
            DataLayer dal = new DataLayer();
            string cid = getCookieID();
            List<Order> opennedOrders = (from x in dal.Orders where (x.CID.Equals(cid) && x.Status.Equals("Openned")) select x).ToList<Order>();
            return Json(opennedOrders, JsonRequestBehavior.AllowGet);
        }
      
        private string getCookieID()
        {
            string cookieName = FormsAuthentication.FormsCookieName; //Find cookie name
            HttpCookie authCookie = HttpContext.Request.Cookies[cookieName]; //Get the cookie by it's name
            FormsAuthenticationTicket ticket = FormsAuthentication.Decrypt(authCookie.Value); //Decrypt it
            return ticket.Name; //You have the UserName!
        }
    }
}