using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Project.Models
{
    public class PartialOrder
    {
        public int OID { get; set; }
        public string Orderdate { get; set; }
        public decimal Totalprice { get; set; }
        
        public PartialOrder(int oid, decimal Total, string orderdate)
        {
            OID = oid;
            Totalprice = Total;
            Orderdate = orderdate;
        }
    }
}