using Project.DAL;
using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Security;

namespace Project.ViewModels
{
    public class OrderVM
    {
        public List<PartialOrder> existOrders { get; set; }
        public OrderVM(string cid)
        {
            existOrders = new List<PartialOrder>();
            setOrders(cid);
        }
        private void setOrders(string cid)
        {
            DataLayer dal = new DataLayer();
            List<Order> Orders = (from x in dal.Orders where (x.CID.Equals(cid) && x.Status.Equals("Openned")) select x).ToList<Order>();
            foreach (Order order in Orders)
            {
                PartialOrder po = new PartialOrder(order.OID, order.Totalprice, order.Orderdate.ToShortDateString());
                existOrders.Add(po);
            }

        }
    }
}