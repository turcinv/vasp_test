from vasp_tools.reaction_time import save_reaction_times

# Sample test data
TEST_TRAJ_FILE = "test_traj.xyz"
TEST_TOPO_FILE = "test_top.pdb"


def test_save_reaction_times():
    """Test that reaction times are computed correctly."""
    reaction_times = save_reaction_times(TEST_TRAJ_FILE, TEST_TOPO_FILE, box_size=13.390, threshold=1.2)

    # Check that reaction_times is a list
    assert isinstance(reaction_times, list)

    # Check if reaction times contain only floats
    assert all(isinstance(rt, float) for rt in reaction_times)

    # Check if reaction times are sorted
    assert reaction_times == sorted(reaction_times)
