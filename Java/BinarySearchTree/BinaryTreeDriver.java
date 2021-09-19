
public class BinaryTreeDriver 
{
	public static void main(String arg[])
	{
		BinarySearchTree b = new BinarySearchTree();
		b.Insert(3);
		b.Insert(8);
		b.Insert(1);
		b.Insert(4);
		b.Insert(6);
		b.Insert(2);
		b.Insert(10);
		b.Insert(9);
		b.Insert(20);
		b.Insert(25);
		b.Insert(15);
		b.Insert(16);
		System.out.println("Original Tree: ");
		b.Iterative_inorder(); //needs to work also with Recursive
		//Inorder function
		
		System.out.println("Find Node with value 4:");
		Node four = b.Search(4);
		if (four != null)
			System.out.println(four.getValue());
		else
			System.out.println("Not found in the tree");
			
		
		System.out.println("Delete Node with no children (2): " +
				b.Delete(2));
		
		b.Iterative_inorder();
		System.out.println("\nDelete Node with one child (4) : " +
			b.Delete(4));
		b.Recursive_inorder();
		System.out.println("\nDelete Node with Two children (10) : " +
			b.Delete(10));
		b.Iterative_inorder();
		
		
		
	}
}
