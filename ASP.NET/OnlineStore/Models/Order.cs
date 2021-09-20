using Project.DAL;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;
using System.Web.Security;

namespace Project.Models
{
    public class Order
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.None)]
        public int OID { get; set; }

       [Required]
        public string CID { get; set; }
        [Required]
        public decimal Totalprice { get; set; }
        [Required]
        public DateTime Orderdate {get;set;}
        [Required]
        public string Status { get; set; }
        public Order()
        { 
        }
        public Order(string cid, decimal totalPrice)
        {
            this.OID = GetNextOrderID();
            this.CID = cid;
            this.Totalprice = totalPrice;
            Orderdate = DateTime.Now.Date;
            Status = "Openned";
        }

        private int GetNextOrderID()
        {
            DataLayer dal = new DataLayer();
            List<Order> orders = dal.Orders.ToList<Order>();
            if (orders.Count > 0)
                return orders[orders.Count-1].OID + 1;
            return 100;
        }
    }
}