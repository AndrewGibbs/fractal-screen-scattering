import numpy as np

def disc(rad,centre,depth, Nr = 400, Nth = 400, Nz = 50):
    #cylinder centre
    c = np.array(centre)
    #cylinder depth
    d = depth

    #now meshgrid the top and bottom discs
    r = np.linspace(0,rad,num=Nr)
    theta = np.linspace(0,2*np.pi,num=Nth)
    x = c[0] + r*np.cos(theta)
    y = c[1] + r*np.sin(theta)
    top_z = c[2] + d/2
    bot_z = c[2] - d/2
    [xx_discs, yy_discs, zz_discs] = np.meshgrid(x,y,[top_z,bot_z])
#    [xx_bot, yy_bot, zz_bot] = np.meshgrid(x,y,[bot_z])

    #now meshgrid the circular edge
    x_rad = c[0] + rad*np.cos(theta)
    y_rad = c[1] + rad*np.sin(theta)
    z = np.linspace(bot_z,top_z,Nz)
    [xx_edge, zz_edge] = np.meshgrid(x_rad,z)
    [yy_edge, _] = np.meshgrid(y_rad,z)

    #piece it all together
    XX = np.hstack([xx_discs.ravel(), xx_edge.ravel()])
    YY = np.hstack([yy_discs.ravel(), yy_edge.ravel()])
    ZZ = np.hstack([zz_discs.ravel(), zz_edge.ravel()])

    points = np.array([XX, YY, ZZ])

    return points

def cuboid(side, centre, depth, Nxy = 10, Nz = 5):
    #cube centre
    c = np.array(centre)
    #cube depth
    d = depth

    #make limits of cuboid
    z_lim = [c[2]-d/2, c[2]+d/2]
    x_lim = [c[0]-side/2, c[0]+side/2]
    y_lim = [c[1]-side/2, c[1]+side/2]
    
    x = np.linspace(x_lim[0],x_lim[1],num=Nxy)
    y = np.linspace(y_lim[0],y_lim[1],num=Nxy)
    z = np.linspace(z_lim[0],z_lim[1],num=Nz)
    [Zxx,Zyy,Zzz] = np.meshgrid(x,y,z_lim)
    [Yxx,Yyy,Yzz] = np.meshgrid(x,y_lim,z)
    [Xxx,Xyy,Xzz] = np.meshgrid(x_lim,y,z)

    XX = np.hstack([Zxx.ravel(), Yxx.ravel(), Xxx.ravel()])
    YY = np.hstack([Zyy.ravel(), Yyy.ravel(), Xyy.ravel()])
    ZZ = np.hstack([Zzz.ravel(), Yzz.ravel(), Xzz.ravel()])

    points = np.array([XX, YY, ZZ])

    return points