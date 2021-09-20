using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;

namespace Project.Models
{
    public class OrderDescription
    {
        [Key]
        [Column(Order = 1)]
        [DatabaseGenerated(DatabaseGeneratedOption.None)]
        public int OID { get; set; }
        [Key]
        [Column(Order = 2)]
        [DatabaseGenerated(DatabaseGeneratedOption.None)]
        public int PID { get; set; }
        [Required]
        public int Amount { get; set; }
        public OrderDescription()
        {
        }
        public OrderDescription(int oid, int PID, int amount)
        {
            this.OID = oid;
            this.PID = PID;
            this.Amount = amount;
        }
    }
}